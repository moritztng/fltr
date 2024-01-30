use clap::{Parser, Subcommand};
use mixtral::Model;
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    io::{prelude::*, BufReader},
    net::TcpListener,
    path::Path,
};
use url::Url;

#[derive(Deserialize)]
struct Prompt {
    name: String,
    prefix: String,
    postfix: String,
    output_len: usize,
}

#[derive(Deserialize)]
struct Server {
    weights: String,
    port: u16,
    prompts: Vec<Prompt>,
}

#[derive(Deserialize)]
struct Config {
    server: Server,
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
        #[arg(long)]
        weights: String,
        #[arg(long, value_delimiter = ',')]
        prompts: Vec<String>,
        #[arg(long, default_value_t = 256)]
        length: usize,
        #[arg(long)]
        autostop: bool,
    },
    Server,
}

fn main() {
    let args = Args::parse();
    match args.command {
        Commands::Generate {
            weights,
            prompts,
            length,
            autostop,
        } => {
            let mut model = Model::from_dir(Path::new(&weights));
            model.generate(&prompts, length - 1, true, autostop, None);
        }
        Commands::Server => {
            let config: Config =
                toml::from_str(&fs::read_to_string("config.toml").unwrap()).unwrap();
            let mut model = Model::from_dir(Path::new(&config.server.weights));
            let mut prompts = HashMap::new();
            for prompt in config.server.prompts {
                let (cache, _) = model.generate(&[prompt.prefix], 0, true, false, None);
                prompts.insert(
                    prompt.name,
                    (
                        cache,
                        prompt.postfix,
                        prompt.output_len,
                    ),
                );
            }
            let listener = TcpListener::bind(("127.0.0.1", config.server.port)).unwrap();
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut buffer = [0u8; 100000];
                loop {
                    let mut headers = [httparse::EMPTY_HEADER; 64];
                    let mut request = httparse::Request::new(&mut headers);
                    reader.read(&mut buffer).unwrap();
                    if request.parse(&buffer).unwrap().is_complete() {
                        let url_parts: Vec<&str> = request.path.unwrap().split('?').collect();
                        let mut url = Url::from_file_path(url_parts[0]).unwrap();
                        url.set_query(Some(url_parts[1]));
                        let query_args: HashMap<_, _> = url.query_pairs().collect();
                        let (cache, postfix, output_len) = prompts
                            .get(&query_args.get("prompt").unwrap().to_string())
                            .unwrap();
                        let input = query_args.get("input").unwrap().to_string();
                        let prompts: Vec<_> = input.split("<inputsep>").map(|x|x.to_owned() + postfix).collect();
                        let (_, outputs) = model.generate(&prompts, *output_len - 1, true, false, Some(cache));
                        let response = format!("HTTP/1.1 200 OK\r\n\r\n{}", outputs.join(","));
                        stream.write_all(response.as_bytes()).unwrap();
                        break;
                    }
                }
            }
        }
    }
}
