from PIL import Image
def cut_3_white(img):
    new_img = Image.new("RGB", img.size)
    img = img.crop((0, 0, 639, 480))
    print(img.size)
    new_image = new_img.paste(img)

    return new_img

if __name__ == "__main__":
    for i in range(100, 550):
        d = "/home/werka/Documents/Python/Python_project/data/train/peace/peace_"+str(i)+".jpg"
        img = img = Image.open(d)
        print(img)
        print(img.size)
        new_image = cut_3_white(img)
        ind = i+500
        dir = "data_a/1_3/peace/peace_"+str(ind)+".jpg"
        new_image.save(dir, quality=100)


