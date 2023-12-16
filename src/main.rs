use clap::Parser;
use llamars::generate;
use std::error::Error;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    weights: String,

    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value_t = 256)]
    length: usize,
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();
    generate(args.weights, args.prompt, args.length, true)?;
    Ok(())
}
