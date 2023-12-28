use clap::{Parser, Subcommand};
use llamars::Model;
use serde::Deserialize;
use std::{
    io::{prelude::*, BufReader},
    net::TcpListener, fs, collections::HashMap,
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
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        length: usize,
        #[arg(long)]
        autostop: bool,
    },
    Server
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Generate {
            weights,
            prompt,
            length,
            autostop,
        } => {
            let mut model = Model::from_file(&weights);
            model.generate(&prompt, length, true, autostop, None).unwrap();
        }
        Commands::Server => {
            let config: Config = toml::from_str(&fs::read_to_string("config.toml").unwrap()).unwrap();
            let mut model = Model::from_file(&config.server.weights);
            let mut prompts = HashMap::new();
            for prompt in config.server.prompts {
                prompts.insert(prompt.name, (model.compile(&prompt.prefix), prompt.postfix, prompt.output_len));
            }
            let listener = TcpListener::bind(("127.0.0.1", config.server.port)).unwrap();
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut buffer = [0u8; 1024];
                let mut output: Option<String> = None;
                loop {
                    let mut headers = [httparse::EMPTY_HEADER; 64];
                    let mut request = httparse::Request::new(&mut headers);
                    reader.read(&mut buffer).unwrap();
                    if request.parse(&buffer).unwrap().is_complete() {
                        let url_parts: Vec<&str> = request.path.unwrap().split('?').collect();
                        let mut url = Url::from_file_path(url_parts[0]).unwrap();
                        url.set_query(Some(url_parts[1]));
                        let query_args: HashMap<_, _> = url
                            .query_pairs()
                            .collect();
                        let (cache, postfix, output_len) = prompts.get(&query_args.get("prompt").unwrap().to_string()).unwrap();
                        let input = query_args.get("input").unwrap().to_string() + postfix;
                        output = Some(model.generate(&input, *output_len, true, true, Some(cache)).unwrap());
                        break;
                    }
                }
                let response = if let Some(output) = output {
                    format!("HTTP/1.1 200 OK\r\n\r\n{output}")
                } else {
                    "HTTP/1.1 404 NOT FOUND".to_owned()
                };
                stream.write_all(response.as_bytes()).unwrap();
            }
        }
    }
}
