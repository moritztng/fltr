use llamars::generate;
use std::error::Error;

#[test]
fn test_generate() -> Result<(), Box<dyn Error + Send + Sync>> {
    assert_eq!(
        generate("weights.bin".into(), "Once upon a time".into(), 14, true)?,
        "<s> Once upon a time, there was a little girl named Lily."
    );
    Ok(())
}
