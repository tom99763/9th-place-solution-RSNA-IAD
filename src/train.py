import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Your main training function.
    The 'cfg' object contains all your configuration.
    """
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    # --- Access config values ---
    print(f"Learning Rate: {cfg.optimizer.lr}")
    print(f"Training for {cfg.training.epochs} epochs.")

    # --- Your ML Logic Goes Here ---
    # Example:
    # model = build_model(cfg.model)
    # optimizer = get_optimizer(cfg.optimizer)
    # run_training_loop(model, optimizer, epochs=cfg.training.epochs)
    
    print("\n✅ Training script finished.")



if __name__ == "__main__":
    train()
