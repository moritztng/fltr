use std::path::Path;
use mixtral::Model;

#[test]
fn test_generate() {
    let mut model = Model::from_dir(Path::new("models/mixtral"));
    model.generate(&"Once upon a time,".to_owned(), 10, true, false, None);
}
