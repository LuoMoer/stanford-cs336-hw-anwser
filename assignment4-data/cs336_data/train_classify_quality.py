import fasttext
model = fasttext.train_supervised(input="./data/classify_quality.train")
print(model.test("./data/classify_quality.valid"))

model.save_model("./classifiers/quality.bin")