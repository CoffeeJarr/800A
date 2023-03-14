class CUDACIFAR10(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            to_cuda: bool = True,
            half: bool = False,
            pre_transform: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False) -> None:

        super().__init__(root, train, transform, target_transform, download)
        if pre_transform is not None:
            self.data = self.data.astype("float32")
            for index in range(len(self)):
                """
                ToTensor的操作会检查数据类型是否为uint8, 如果是, 则除以255进行归一化, 这里data提前转为float,
                所以手动除以255.
                """
                self.data[index] = pre_transform(self.data[index]/255.0).numpy().transpose((1, 2, 0))
                self.targets[index] = torch.Tensor([self.targets[index]]).squeeze_().long()
                if to_cuda:
                    self.targets[index] = self.targets[index].cuda()
            self.data = torch.Tensor(self.data).permute((0, 3, 1, 2))
            if half:
                self.data = self.data.half()
            if to_cuda:
                self.data = self.data.cuda()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target