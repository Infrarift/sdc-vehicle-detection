from trainer import Trainer


def load_training_data():
    training_path = "../training_data/"
    trainer = Trainer(training_path)
    trainer.load_shuffle_split_join("vehicles/", 1)
    trainer.load_shuffle_split_join("non-vehicles/", 0)
    print("Data loading complete: {0} Training Features, {1} Testing Features"
          .format(len(trainer.train_features), format(len(trainer.test_features))))

if __name__ == "__main__":
    load_training_data()
    pass
