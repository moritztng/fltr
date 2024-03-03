use std::path::Path;
use fltr::Model;

#[test]
fn test_generate() {
    let mut model = Model::from_dir(Path::new("/usr/share/fltr"));
    model.generate(&vec!["Once upon a time,".to_owned()], 10, true, false, None);
}
