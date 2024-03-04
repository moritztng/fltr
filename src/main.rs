use clap::{Parser, Subcommand};
use fltr::Model;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None, subcommand_negates_reqs = true)]
struct Args {
    #[arg(long, required = true)]
    file: Option<String>,
    #[arg(long, required = true)]
    prompt: Option<String>,
    #[arg(long, default_value = "/usr/share/fltr")]
    weights: Option<String>,
    #[arg(long, default_value = "1")]
    batch_size: Option<usize>,
    #[arg(long)]
    debug: bool,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(long, default_value = "/usr/share/fltr")]
        weights: String,
        #[arg(long, value_delimiter = ',', required = true)]
        prompts: Vec<String>,
        #[arg(long, default_value_t = 256)]
        length: usize,
        #[arg(long)]
        autostop: bool,
    },
}

fn main() {
    let args = Args::parse();
    if let Some(Commands::Generate {
        weights,
        prompts,
        length,
        autostop,
    }) = args.command
    {
        let mut model = Model::from_dir(Path::new(&weights));
        model.generate(&prompts, length - 1, true, autostop, None);
    } else {
        let mut model = Model::from_dir(Path::new(&args.weights.unwrap()));
        let (cache, _) = model.generate(
            &[format!("[INST] {}", args.prompt.unwrap())],
            0,
            args.debug,
            false,
            None,
        );
        let mut batch = Vec::new();
        let mut filter_batch = |batch: &Vec<String>| {
            let prompts: Vec<String> = batch
                .iter()
                .map(|x| x.to_owned() + "\nAnswer: Yes or No\nAnswer:[/INST]")
                .collect();
            let (_, outputs) = model.generate(&prompts, 0, args.debug, false, Some(&cache));
            batch
                .iter()
                .zip(outputs)
                .filter(|(_, output)| output.to_lowercase() == "yes")
                .for_each(|(x, _)| println!("{}", x));
        };
        for line in BufReader::new(File::open(args.file.unwrap()).unwrap()).lines() {
            batch.push(line.unwrap());
            if batch.len() == args.batch_size.unwrap() {
                filter_batch(&batch);
                batch.clear();
            }
        }
        if !batch.is_empty() {
            filter_batch(&batch);
        }
    }
}
