from customtkinter import CTk, CTkFrame, CTkEntry, CTkLabel, CTkButton, CTkImage, StringVar
from image_continuation import Painter, get_data_from_image, Image
from tkinter import filedialog
import sys

print(sys.argv)
current_path = sys.argv[0]
sample_path = "./sample1.png"
if len(sys.argv) > 1:
    sample_path = sys.argv[1]
sample_image = Image.open(sample_path).convert(mode="RGB")
test_image = Image.new("RGB", (50, 50))


def str_tuple(text, _type=float):
    li = [""]
    for ch in text:
        if ch in "0987654321.":
            li[-1] += ch
        elif ch == ",":
            li.append("")
    try:
        return tuple(map(_type, li))
    except ValueError:
        return False


class LabeledEntry(CTkFrame):
    def __init__(self, master, label, place_holder="", width=200, **kwargs):
        super(LabeledEntry, self).__init__(master)
        self.label = CTkLabel(self, width=width, text=label)
        self.value = StringVar(self, place_holder)
        self.entry = CTkEntry(self, textvariable=self.value, placeholder_text=place_holder)
        self.label.pack(), self.entry.pack(fill="x")
        self.pack(fill="x", padx=5, pady=5, **kwargs)

    def get(self):
        return self.entry.get()


class App(CTk):
    def __init__(self):
        super(App, self).__init__()
        self.net = Painter(5, 30)
        self.data = get_data_from_image(sample_image)

        # UI
        self.title("Image Continuation")
        self.main_frames = [CTkFrame(self), CTkFrame(self), CTkFrame(self)]
        for frame in self.main_frames:
            frame.pack(side="left", expand=True, fill="both")

        self.image_path = LabeledEntry(self.main_frames[0], "Sample image path", sample_path)
        self.choose_file = CTkButton(self.main_frames[0], 100, text="Open file", command=self.open)
        self.choose_file.pack()
        self.x_range = LabeledEntry(self.main_frames[0], "X-axis range", "(0, 1)")
        self.y_range = LabeledEntry(self.main_frames[0], "Y-axis range", "(0, 1)")
        self.net_struct = LabeledEntry(self.main_frames[0], "Net structure", "(5, 30)")
        self.reset_button = CTkButton(self.main_frames[0], text="Reset", width=60, command=self.reset)
        self.reset_button.pack()

        self.training_times = LabeledEntry(self.main_frames[1], "Train for", "100")
        buttons_frame = CTkFrame(self.main_frames[1])
        self.train_button = CTkButton(buttons_frame, text="Train", width=60, command=self.train)
        self.test_button = CTkButton(buttons_frame, text="Test", width=60, command=self.test)
        self.train_button.pack(side="left", padx=5), self.test_button.pack(side="left", padx=5), buttons_frame.pack()
        self.image_size = LabeledEntry(self.main_frames[1], "Output image size", "(200, 200)")
        # self.out_image_path = LabeledEntry(self.main_frames[1], "Output image path", "./Output image.png")
        self.generate_button = CTkButton(self.main_frames[1], text="Generate image", command=self.generate)
        self.generate_button.pack()

        self.sample_label = CTkLabel(self.main_frames[2], text="Sample:")
        self.sample_image = CTkLabel(self.main_frames[2], text="",
                                     image=CTkImage(sample_image).create_scaled_photo_image(10, "light"))
        self.test_label = CTkLabel(self.main_frames[2], text="Test:")
        self.test_image = CTkLabel(self.main_frames[2], text="",
                                   image=CTkImage(test_image).create_scaled_photo_image(10, "light"))
        self.sample_label.pack(), self.sample_image.pack(padx=10), self.test_label.pack(), self.test_image.pack(padx=10)

    def open(self):
        global sample_path
        path = filedialog.Open(self).show()
        if path != "":
            sample_path = path
            self.image_path.value.set(sample_path)

    def reset(self):
        global sample_image
        img_path, net_str, xr, yr = self.image_path.get(), self.net_struct.get(), self.x_range.get(), self.y_range.get()
        net_str = str_tuple(net_str, int) or (5, 30)
        xr, yr = str_tuple(xr) or (0, 1), str_tuple(yr) or (0, 1)
        self.net = Painter(*net_str)
        if img_path != "":
            sample_image = Image.open(img_path).convert(mode="RGB")
            self.sample_image.configure(image=CTkImage(sample_image).create_scaled_photo_image(10, "light"))
        self.data = get_data_from_image(sample_image, xr, yr)

    def train(self):
        time = self.training_times.get()
        self.net.train(self.data, int(time or 100))

    def test(self):
        self.net.paint(test_image)
        self.test_image.configure(image=CTkImage(test_image).create_scaled_photo_image(10, "light"))

    def generate(self):
        size = self.image_size.get()
        location = filedialog.SaveAs(self).show()
        if location != "":
            img = self.net.paint(Image.new("RGB", str_tuple(size, int) or (200, 200)))
            img.show(), img.save(location)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
