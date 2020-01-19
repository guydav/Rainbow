import numpy as np
import torch
from collections import namedtuple

MaskerDefinition = namedtuple('MaskerDefinition', ('filter_colors', 'row_range', 'col_range', 'range_whitelist'))
MaskerDefinition.__new__.__defaults__ = (None, (None, None), (None, None), False)


FULL_FRAME_SHAPE = (210, 160)
SMALL_FRAME_SHAPE = (84, 84)

player_colors = ((162, 98, 33),
                 (162, 162, 42),
                 (198, 108, 58),
                 (142, 142, 142) # also captures the igloo
                )

unvisited_floe_colors = ((214, 214, 214), # unvisited_floes
                        )
visited_floe_colors = ((84, 138, 210), # visited floes
                      )
land_colors = ((192, 192, 192), # the lighter ground in earlier levels
               (74, 74, 74), # the darker ground in later levels
              )

land_row_min = 42
land_row_max = 78
land_row_range = (land_row_min, land_row_max)

bad_animal_colors = ((132, 144, 252), # birds -- also captures the score!
                     (213, 130, 74), # crabs -- no more conflict with the player
                     (210, 210, 64), # angry yellow things
                    )

bear_colors = ((111, 111, 111), # bear in white background
               (214, 214, 214), # bear in black background -- same as the unvisited floes
              )

good_animal_colors = ((111, 210, 111), # fish
                     )

animal_full_frame_row_min = 78
animal_full_frame_row_max = 185
animal_full_frame_row_range = (animal_full_frame_row_min, animal_full_frame_row_max)

igloo_colors = ((142, 142, 142),
                # isolating the igloo door is harder - its black and orange colors both conflict
                )

igloo_full_frame_row_min = 35
igloo_full_frame_row_max = 55
igloo_full_frame_row_range = (igloo_full_frame_row_min, igloo_full_frame_row_max)

igloo_full_frame_col_min = 112
igloo_full_frame_col_max = 144
igloo_full_frame_col_range = (igloo_full_frame_col_min, igloo_full_frame_col_max)


class ColorFilterMasker:
  def __init__(self, masker_def):
    self.filter_colors = np.stack([np.tile(np.array(x, dtype=np.uint8).reshape(1, 1, 3), (*FULL_FRAME_SHAPE, 3))
                                   for x in masker_def.filter_colors], axis=3)
    self.row_range = masker_def.row_range
    self.col_range = masker_def.col_range
    self.range_whitelist = masker_def.range_whitelist

  def __call__(self, frame):
    """
    Assumes the frame is of the form [h, w, c]
    """
    # TODO: reimplement this natively with torch operations
    # TODO: afterwards, check which part of this function is actually the slow part -- the mask or the whitelisting
    mask = np.any(np.all(np.equal(np.expand_dims(frame, 3), self.filter_colors), axis=2), axis=2).astype(np.uint8)

    if self.range_whitelist:
      mask[np.r_[0:self.row_range[0], self.row_range[1]:mask.shape[0]], :] = 0
      mask[:, np.r_[0:self.col_range[0], self.col_range[1]:mask.shape[1]]] = 0

    else:
      mask[self.row_range[0]:self.row_range[1], self.col_range[0]:self.col_range[1]] = 0

    return mask


class TorchMasker:
    def __init__(self, masker_definitions, device, zero_mask_indices=None):
        self.masker_definitions = sorted(list(masker_definitions), key=lambda md: len(md.filter_colors), reverse=True)
        self.device = device

        all_colors = []
        for masker in self.masker_definitions:
            all_colors.extend(masker.filter_colors)

        self.all_colors = torch.stack([torch.tensor(color, dtype=torch.float, device=device).view(1, 1, 3).repeat(*FULL_FRAME_SHAPE, 1)
                                       for color in all_colors])
        self.category_lengths = [len(masker.filter_colors) for masker in self.masker_definitions]
        
        self.zero_mask = torch.ones(len(self.masker_definitions), *FULL_FRAME_SHAPE, device=self.device)
        for i, masker_def in enumerate(self.masker_definitions):
            if zero_mask_indices is not None and i in zero_mask_indices:
                self.zero_mask[i, :, :] = 0

            elif masker_def.range_whitelist:
                self.zero_mask[i, :masker_def.row_range[0], :] = 0
                self.zero_mask[i, masker_def.row_range[1]:, :] = 0
                self.zero_mask[i, :, :masker_def.col_range[0]] = 0
                self.zero_mask[i, :, masker_def.col_range[1]:] = 0

            else:
                self.zero_mask[i, masker_def.row_range[0]:masker_def.row_range[1],
                               masker_def.col_range[0]:masker_def.col_range[1]] = 0

    def __call__(self, frame):
        all_mask_results = torch.eq(frame.view(1, *frame.shape), self.all_colors).all(dim=3)
        category_masks = torch.zeros(len(self.masker_definitions), *FULL_FRAME_SHAPE, device=self.device)

        current_index = 0
        for i, length in enumerate(self.category_lengths):
            if length > 1:
                category_masks[i] = all_mask_results[current_index: current_index + length].any(dim=0)
                current_index += length
            else:
                category_masks[i:] = all_mask_results[current_index:]
                break

        category_masks.mul_(self.zero_mask)

        return category_masks    
# Below natively with uint8s -- it seems almost identical, perhaps a second slower over 20k steps
"""
class TorchMasker:
    def __init__(self, masker_definitions, device):
        self.masker_definitions = sorted(list(masker_definitions), key=lambda md: len(md.filter_colors), reverse=True)
        self.device = device

        all_colors = []
        for masker in self.masker_definitions:
            all_colors.extend(masker.filter_colors)

        # TODO: profile torch.uint8 here vs. converting to float later
        # self.all_colors = torch.stack([torch.tensor(color, dtype=torch.float, device=device).view(1, 1, 3).repeat(*FULL_FRAME_SHAPE, 1)
        self.all_colors = torch.stack([torch.tensor(color, dtype=torch.uint8, device=device).view(1, 1, 3).repeat(*FULL_FRAME_SHAPE, 1)
                                       for color in all_colors])
        self.category_lengths = [len(masker.filter_colors) for masker in self.masker_definitions]

    def __call__(self, frame):
        all_mask_results = torch.eq(frame.view(1, *frame.shape).type(torch.uint8), self.all_colors).all(dim=3)
        category_masks = torch.zeros(len(self.masker_definitions), *FULL_FRAME_SHAPE, device=self.device, dtype=torch.uint8)

        current_index = 0
        for i, length in enumerate(self.category_lengths):
            if length > 1:
                category_masks[i] = all_mask_results[current_index: current_index + length].any(dim=0)
                current_index += length
            else:
                category_masks[i:] = all_mask_results[current_index:]
                break

        for i, masker_def in enumerate(self.masker_definitions):
            if masker_def.range_whitelist:
                category_masks[i, :masker_def.row_range[0], :] = 0
                category_masks[i, masker_def.row_range[1]:, :] = 0
                category_masks[i, :, :masker_def.col_range[0]] = 0
                category_masks[i, :, masker_def.col_range[1]:] = 0

            else:
                category_masks[i, masker_def.row_range[0]:masker_def.row_range[1],
                               masker_def.col_range[0]:masker_def.col_range[1]] = 0

        return category_masks.float()
"""


# player_masker = ColorFilterMasker(player_colors, igloo_full_frame_row_range,
#                                   igloo_full_frame_col_range)
# unvisited_floe_masker = ColorFilterMasker(unvisited_floe_colors, animal_full_frame_row_range,
#                                           (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
# visited_floe_masker = ColorFilterMasker(visited_floe_colors, animal_full_frame_row_range,
#                                         (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
# land_masker = ColorFilterMasker(land_colors, land_row_range,
#                                 (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
# bad_animal_masker = ColorFilterMasker(bad_animal_colors, animal_full_frame_row_range,
#                                       (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
# good_animal_masker = ColorFilterMasker(good_animal_colors, animal_full_frame_row_range,
#                                        (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
# bear_filter = ColorFilterMasker(bear_colors, (0, animal_full_frame_row_min),
#                                 (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
# igloo_masker = ColorFilterMasker(igloo_colors, igloo_full_frame_row_range,
#                                  igloo_full_frame_col_range, range_whitelist=True)

player_masker_def = MaskerDefinition(player_colors, igloo_full_frame_row_range,
                                     igloo_full_frame_col_range)
unvisited_floe_masker_def = MaskerDefinition(unvisited_floe_colors, animal_full_frame_row_range,
                                             (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
visited_floe_masker_def = MaskerDefinition(visited_floe_colors, animal_full_frame_row_range,
                                           (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
land_masker_def = MaskerDefinition(land_colors, land_row_range,
                                   (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
bad_animal_masker_def = MaskerDefinition(bad_animal_colors, animal_full_frame_row_range,
                                         (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
good_animal_masker_def = MaskerDefinition(good_animal_colors, animal_full_frame_row_range,
                                          (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
bear_filter_def = MaskerDefinition(bear_colors, (0, animal_full_frame_row_min),
                                   (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
igloo_masker_def = MaskerDefinition(igloo_colors, igloo_full_frame_row_range,
                                    igloo_full_frame_col_range, range_whitelist=True)

ALL_MASKERS = {
    'player': player_masker_def,
    'unvisited_floe': unvisited_floe_masker_def,
    'visited_floe': visited_floe_masker_def,
    'land': land_masker_def,
    'bad_animal': bad_animal_masker_def,
    'good_animal': good_animal_masker_def,
    'bear': bear_filter_def,
    'igloo': igloo_masker_def
}
