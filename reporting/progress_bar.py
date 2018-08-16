import progressbar


class ProgressBar(object):
    def __init__(self, output_format, value_dict, total):
        self.value_dict = value_dict
        self.format_custom_text = progressbar.FormatCustomText(
            output_format,
            value_dict
        )
        self.prog_bar = progressbar.ProgressBar(max_value=total, widgets=[
            progressbar.Counter(), ' of {} '.format(total),
            progressbar.Bar(),
            ' ', progressbar.ETA(),
            ' ', self.format_custom_text
        ])

    def update(self, index, new_dict):
        self.format_custom_text.update_mapping(loss=new_dict['loss'])
        self.prog_bar.update(index)

    def finish(self):
        self.prog_bar.finish()
