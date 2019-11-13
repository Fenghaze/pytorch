from tensorboard_logger import Logger

logger = Logger(logdir='experiment_cnn', flush_secs=2)

for i in range(100):
    logger.log_value('loss', 10-i**0.5, step=i)
    logger.log_value('accuracy', i**0.5/10)