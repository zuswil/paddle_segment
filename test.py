class cal():
    def __init__(self, num):
        self.num = num

    def forward(self, num):
        self.num = self.num + 1
        return self.num


class test(cal):
    def __init__(self, num):
        super(test, self).__init__(
            num
        )

if __name__ == '__main__':
    t = test(3)
    print(t)