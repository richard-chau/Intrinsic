class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.inres = nn.Conv2d(num_classes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnets = list(models.resnet50(pretrained=False).children())
        self.res_o16 = nn.Sequential(*self.resnets[1:-3])
        self.last1 = nn.Sequential(
            nn.Conv2d(256, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            )
        self.last2 = nn.Conv2d(256, 16, 3, padding=1, bias=False) nn.Sequential(
            nn.Conv2d(256, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            )
        

    def forward(self, x):
        inres = self.inres(x)
        resout = self.res_o16(inres)
        out1 = self.last1(resout)
        out2 = self.last2(resout)
        return F.upsample_bilinear(out1, x,size()[2:]), F.upsample_bilinear(out2, x,size()[2:])
