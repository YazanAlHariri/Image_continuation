from PIL import Image
from NeuralNet import Network


def get_data_from_image(img: Image.Image, x_range=(0, 1), y_range=(0, 1)):
    data = []
    dx, dy = (x_range[1] - x_range[0]) / img.width, (y_range[1] - y_range[0]) / img.height
    for a in range(img.height * img.width):
        xy = divmod(a, img.height)
        c = tuple(map(lambda x: x / 255, img.getpixel(xy)))
        if c != (0, 0, 0):
            data.append(((xy[0] * dx + x_range[0], xy[1] * dy + y_range[0]), c))
    return data


class Painter(Network):
    def __init__(self, hidden_layers, neurons_each):
        super(Painter, self).__init__()
        self.initialize(2, hidden_layers, neurons_each, 3)

    def paint(self, img) -> Image.Image:
        for x in range(img.width):
            for y in range(img.height):
                color = tuple(map(lambda c: int(c * 255), self.evaluate((x / img.width, y / img.height))))
                img.putpixel((x, y), color)
        return img

    def train(self, data, intensity=200):
        for _ in range(intensity):
            for inp, target in data:
                self.backpropagation(inp, target)


def main():
    with Image.open("./sample.png").convert(mode="RGB") as img:
        data = get_data_from_image(img, (0.2, 0.8), (0.2, 0.8))
        print(len(data), data)
        painter = Painter(5, 30)
        # net.learning_rate *= 4
        while True:
            inp = input("Train: ")
            if inp == "No":
                break
            painter.train(data, int(inp))
            painter.paint(img)
            img.show()
        painter.paint(Image.new("RGB", (200, 200))).show()


if __name__ == '__main__':
    main()
