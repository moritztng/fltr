#[cfg(feature = "cuda")]
use cc;
fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/kernels.cu");
        cc::Build::new()
            .cuda(true)
            .file("src/kernels.cu")
            .compile("kernels");
        println!("cargo:rustc-link-lib=cublas");
    }
}
