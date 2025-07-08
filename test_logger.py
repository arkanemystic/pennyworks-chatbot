from utils.logger import setup_logger

def main():
    logger = setup_logger()
    logger.debug("This is a DEBUG message from pennyworks logger.")
    logger.info("This is an INFO message from pennyworks logger.")
    logger.warning("This is a WARNING message from pennyworks logger.")

if __name__ == "__main__":
    main()
