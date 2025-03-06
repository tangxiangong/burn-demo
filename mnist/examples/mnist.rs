use burn::{
    backend::{Autodiff, wgpu},
    optim::AdamConfig,
};
use mnist::{
    model::ModelConfig,
    training::{TrainingConfig, train},
};

fn main() {
    type MyBackend = wgpu::Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = wgpu::WgpuDevice::default();
    println!("Device: {:?}", device);

    let artifact_dir = "artifacts";

    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
