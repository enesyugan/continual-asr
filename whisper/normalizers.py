from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar

class ArNormalizer:
    def __init__(self, norm_unicode=False, norm_orthographic=False, remove_diacrits=False):
        self.norm_unicode = norm_unicode
        self.norm_orthographic = norm_orthographic
        self.remove_diacrits = remove_diacrits
    
    def __call__(self, text):
        if self.norm_unicode:
            text = normalize_unicode(text)
        if self.norm_orthographic:
            text = normalize_alef_maksura_ar(text)
            text = normalize_alef_ar(text)
            text = normalize_teh_marbuta_ar(text)
        if self.remove_diacrits:
            text = dediac_ar(text)
        return text

