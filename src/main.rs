use clap::Parser;
use fltr::Model;
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Parser)]
#[command(about, version, author)]
struct Args {
    #[arg(long)]
    file: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value = "32")]
    batch_size: Option<usize>,
    #[arg(long)]
    large: bool,
    #[arg(long)]
    debug: bool,
}

fn main() {
    let args = Args::parse();
    let mut model = Model::from_dir(Path::new(&env::var("HOME").unwrap()).join("Fltr").as_path(), args.large);
    let mut batch = Vec::new();
    let mut filter_batch = |batch: &Vec<String>| {
        let prompts: Vec<String> = batch
            .iter()
            .map(|x| format!("[INST] {}{}\nAnswer: Yes or No\nAnswer:[/INST]", args.prompt.clone(), x))
            .collect();
        batch
            .iter()
            .zip(model.generate(&prompts,args.debug))
            .filter(|(_, output)| output.to_lowercase() == "yes")
            .for_each(|(x, _)| println!("{}", x));
    };
    for line in BufReader::new(File::open(args.file).unwrap()).lines() {
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
