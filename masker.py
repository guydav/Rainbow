import numpy as np


class ColorFilterMasker:
  def __init__(self, filter_colors, row_range=(None, None), col_range=(None, None), range_whitelist=False):
    self.filter_colors = np.stack([np.array(x).reshape(1, 1, 3) for x in filter_colors], axis=3)
    self.row_range = row_range
    self.col_range = col_range
    self.range_whitelist = range_whitelist

  def __call__(self, frame):
    """
    Assumes the frame is of the form [h, w, c]
    """
    mask = np.any(np.all(np.equal(np.expand_dims(frame, 3), self.filter_colors), axis=2), axis=2).astype(np.uint8)

    if self.range_whitelist:
      mask[np.r_[0:self.row_range[0], self.row_range[1]:mask.shape[0]], :] = 0
      mask[:, np.r_[0:self.col_range[0], self.col_range[1]:mask.shape[1]]] = 0

    else:
      mask[self.row_range[0]:self.row_range[1], self.col_range[0]:self.col_range[1]] = 0

    return mask


full_frame_shape = (210, 160)

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


player_masker = ColorFilterMasker(player_colors, igloo_full_frame_row_range,
                                  igloo_full_frame_col_range)
unvisited_floe_masker = ColorFilterMasker(unvisited_floe_colors, animal_full_frame_row_range,
                                          (0, full_frame_shape[1]), range_whitelist=True)
visited_floe_masker = ColorFilterMasker(visited_floe_colors, animal_full_frame_row_range,
                                        (0, full_frame_shape[1]), range_whitelist=True)
land_masker = ColorFilterMasker(land_colors, land_row_range,
                                 (0, full_frame_shape[1]), range_whitelist=True)
bad_animal_masker = ColorFilterMasker(bad_animal_colors, animal_full_frame_row_range,
                                      (0, full_frame_shape[1]), range_whitelist=True)
good_animal_masker = ColorFilterMasker(good_animal_colors, animal_full_frame_row_range,
                                       (0, full_frame_shape[1]), range_whitelist=True)
bear_filter = ColorFilterMasker(bear_colors, (0, animal_full_frame_row_min),
                                (0, full_frame_shape[1]), range_whitelist=True)
igloo_masker = ColorFilterMasker(igloo_colors, igloo_full_frame_row_range,
                                 igloo_full_frame_col_range, range_whitelist=True)

ALL_MASKERS = {
    'player': player_masker,
    'unvisited_floe': unvisited_floe_masker,
    'visited_floe': visited_floe_masker,
    'land': land_masker,
    'bad_animal': bad_animal_masker,
    'good_animal': good_animal_masker,
    'bear': bear_filter,
    'igloo': igloo_masker
}
