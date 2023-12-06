use llamars::generate;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    generate("weights.bin".into(), "Once upon a time".into(), 256, true)?;
    Ok(())
}
