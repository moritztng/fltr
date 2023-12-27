use clap::{Parser, Subcommand};
use llamars::Model;
use std::{
    io::{prelude::*, BufReader},
    net::TcpListener,
};
use url::Url;

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
    },
    Server {
        #[arg(long)]
        weights: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 5000)]
        port: u16,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Generate {
            weights,
            prompt,
            length,
        } => {
            let mut model = Model::from_file(weights);
            model.generate(prompt, length, true, None).unwrap();
        }
        Commands::Server {
            weights,
            prompt,
            port,
        } => {
            let mut model = Model::from_file(weights);
            let (position, cache) = model.compile(prompt);
            let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
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
                        let parts: Vec<&str> = request.path.unwrap().split('?').collect();
                        let mut url = Url::from_file_path(parts[0]).unwrap();
                        url.set_query(Some(parts[1]));
                        if let Some(prompt) = url
                            .query_pairs()
                            .find_map(
                                |(key, value)| {
                                    if key == "prompt" {
                                        Some(value.to_string())
                                    } else {
                                        None
                                    }
                                },
                            )
                        {
                            output = Some(model.generate(prompt, 10, true, Some((position, &cache))).unwrap());
                        }
                        break;
                    }
                }
                let response = if let Some(output) = output {
                    format!("HTTP/1.1 200 OK\r\n\r\n{output}")
                } else {
                    "HTTP/1.1 404 NOT FOUND".into()
                };
                stream.write_all(response.as_bytes()).unwrap();
            }
        }
    }
}
