use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use mnist::{
    model::ModelConfig,
    training::{TrainingConfig, train},
};

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "artifacts";

    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
