import torch
from . import initialization as init


class MultiBranchModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.cell_decoder)
        init.initialize_head(self.cell_segmentation_head)
        init.initialize_decoder(self.tissue_decoder)
        init.initialize_head(self.tissue_segmentation_head)
        init.initialize_decoder(self.tissue2cell)
        init.initialize_decoder(self.cell2tissue)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x, y):

        h, w = x.shape[-2:]
        output_stride = self.cell_encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

        h, w = y.shape[-2:]
        output_stride = self.tissue_encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    
    # 搭建孪生网络，同时传输两个
    def forward(self, x, y, loc):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x,y)

        cell_features = self.cell_encoder(x)
        tissue_features = self.tissue_encoder(y)
        
        trans1 = self.tissue2cell(tissue_features,cell_features,loc)

        cell_decoder_output = self.cell_decoder(*trans1)
        tissue_decoder_output = self.tissue_decoder(*tissue_features)

        trans2 = self.cell2tissue(tissue_decoder_output,cell_decoder_output,loc)
        cell_masks = self.cell_segmentation_head(cell_decoder_output)
        tissue_masks = self.tissue_segmentation_head(trans2) 

        if self.classification_head is not None:
            labels = self.classification_head(tissue_features[-1])
            return cell_masks, labels
            
        return cell_masks,tissue_masks

    @torch.no_grad()
    def predict(self, x, y, loc):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x,y = self.forward(x,y,loc)

        return x,y
