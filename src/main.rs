use clap::{Parser, Subcommand};
use imap::ClientBuilder;
use llamars::Model;
use mailparse::parse_header;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs, str,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::OpenOptions,
    io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    join,
    sync::mpsc::{self, Receiver, Sender},
    time,
};

#[derive(Deserialize)]
struct EmailConfig {
    domain: String,
    address: String,
    password: String,
    interval: u64,
}

#[derive(Deserialize)]
struct Config {
    email: EmailConfig,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(short, long)]
        weights: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value_t = 256)]
        length: usize,
    },
    Email,
}

#[derive(Serialize, Deserialize, Debug)]
struct Email {
    header: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Event<A> {
    id: u32,
    time: u64,
    data: A,
}

async fn fetch_emails(
    sender: Sender<Email>,
    domain: String,
    address: String,
    password: String,
    interval: u64,
) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("fetch_emails_state.bin")
        .await
        .unwrap();
    let mut last_uid = file.read_u32().await.unwrap_or(0);
    let client = ClientBuilder::new(domain, 993).connect().unwrap();
    let mut imap_session = client.login(address, password).map_err(|e| e.0).unwrap();
    imap_session.select("INBOX").unwrap();
    let mut interval = time::interval(time::Duration::from_secs(interval));
    loop {
        interval.tick().await;
        let messages = imap_session
            .uid_fetch(
                format!("{}:*", last_uid + 1),
                "(BODY[HEADER.FIELDS (SUBJECT)] UID)",
            )
            .unwrap();
        for message in messages.iter() {
            if message.uid.unwrap() == last_uid {
                break;
            }
            let email = Email {
                header: parse_header(message.header().unwrap())
                    .unwrap()
                    .0
                    .get_value(),
            };
            sender.send(email).await.unwrap();
            last_uid = message.uid.unwrap();
        }
        file.seek(io::SeekFrom::Start(0)).await.unwrap();
        file.write_u32(last_uid).await.unwrap();
    }
}

async fn store<A: Serialize + DeserializeOwned + std::fmt::Debug>(
    mut receiver: Receiver<A>,
    sender: Sender<Event<A>>,
    name: String,
) {
    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(format!("{name}_store.bin"))
        .await
        .unwrap();
    let mut state_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(format!("{name}_store_state.bin"))
        .await
        .unwrap();
    let mut id = state_file.read_u32().await.unwrap_or(0);

    while let Some(data) = receiver.recv().await {
        let event = Event {
            id,
            time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            data,
        };
        file.write_all(&bincode::serialize(&event).unwrap())
            .await
            .unwrap();
        sender.send(event).await.unwrap();
        id += 1;
        state_file.seek(io::SeekFrom::Start(0)).await.unwrap();
        state_file.write_u32(id).await.unwrap();
    }
}

async fn map(mut receiver: Receiver<Event<Email>>, sender: Sender<(Event<Email>, bool)>) {
    while let Some(event) = receiver.recv().await {
        sender.send((event, true)).await.unwrap();
    }
}

async fn output<A: std::fmt::Debug + Serialize>(
    mut receiver: Receiver<A>,
    predicate: fn(&A) -> bool,
    name: String,
) {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(format!("{name}.csv"))
        .unwrap();
    let mut csv_writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    while let Some(data) = receiver.recv().await {
        if predicate(&data) {
            csv_writer.serialize(&data).unwrap();
            csv_writer.flush().unwrap();
            println!("{data:?}");
        }
    }
}

#[tokio::main]
async fn main() {
    let config: Config = toml::from_str(&fs::read_to_string("config.toml").unwrap()).unwrap();
    let args = Args::parse();

    match args.command {
        Commands::Generate {
            weights,
            prompt,
            length,
        } => {
            let mut model = Model::from_file(weights);
            model.generate(prompt, length, true).unwrap();
        }
        Commands::Email => {
            let (email_tx, email_rx) = mpsc::channel(5000);
            let (map_tx, map_rx) = mpsc::channel(5000);
            let (output_tx, output_rx) = mpsc::channel(5000);

            let fetch_emails_handle = tokio::spawn(async move {
                fetch_emails(
                    email_tx,
                    config.email.domain,
                    config.email.address,
                    config.email.password,
                    config.email.interval
                )
                .await;
            });
            let store_emails_handle = tokio::spawn(async move {
                store::<Email>(email_rx, map_tx, "emails".into()).await;
            });
            let map_handle = tokio::spawn(async move {
                map(map_rx, output_tx).await;
            });
            let output_handle = tokio::spawn(async move {
                output::<(Event<Email>, bool)>(
                    output_rx,
                    |(_, pred)| *pred,
                    "filtered_emails".into(),
                )
                .await;
            });
            let _ = join!(
                fetch_emails_handle,
                store_emails_handle,
                map_handle,
                output_handle,
            );
        }
    }
}
