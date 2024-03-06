use clap::{Parser, Subcommand};
use fltr::Model;
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Parser)]
#[command(about, version, author, subcommand_negates_reqs = true)]
struct Args {
    #[arg(long, required = true)]
    file: Option<String>,
    #[arg(long, required = true)]
    prompt: Option<String>,
    #[arg(long, default_value = "32")]
    batch_size: Option<usize>,
    #[arg(long)]
    debug: bool,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
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
    let model_path = Path::new(&env::var("HOME").unwrap()).join("Fltr");
    let model_path = model_path.as_path();
    if let Some(Commands::Generate {
        prompts,
        length,
        autostop,
    }) = args.command
    {
        let mut model = Model::from_dir(model_path);
        model.generate(&prompts, length - 1, true, autostop, None);
    } else {
        let mut model = Model::from_dir(model_path);
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
