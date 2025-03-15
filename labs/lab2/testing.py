from treci import ConvolutionalModel



if '__main__' == __name__:
    model = ConvolutionalModel(1,16,32,50,512,10)

    print(model.layers)