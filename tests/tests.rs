use std::path::Path;

use llamars::Model;

#[test]
fn test_generate() {
    let mut model = Model::from_dir(Path::new("models/mistral"));
    model.generate(&"Once upon a time,".to_owned(), 10, true, false, None).unwrap();
}
