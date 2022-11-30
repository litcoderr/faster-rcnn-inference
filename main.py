from model import Extractor


if __name__ == "__main__":
    extractor = Extractor("/faster-rcnn-inference/images", "/faster-rcnn-inference/output")
    extractor.extract()
