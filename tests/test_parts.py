from src.api.parts import split_parts


def test_contrast_after_comma():
    text = 'thanks for the quick help, but I waited two hours on hold'
    assert split_parts(text) == ['thanks for the quick help', 'but I waited two hours on hold']


def test_contrast_after_hyphen():
    text = 'Bad experience today-but the cabin crew was so humble'
    assert split_parts(text) == ['Bad experience today', 'but the cabin crew was so humble']


def test_sentences_are_split():
    text = 'Flight 482 now boards at gate B12. The crew was lovely. Still no bag after 3 days.'
    assert split_parts(text) == [
        'Flight 482 now boards at gate B12.',
        'The crew was lovely.',
        'Still no bag after 3 days.',
    ]


def test_single_part_stays_single():
    assert split_parts('worst flight ever') == ['worst flight ever']


def test_one_word_piece_is_joined_to_previous():
    assert split_parts('Great job on the landing. Thanks.') == ['Great job on the landing. Thanks.']


def test_decimal_is_not_a_sentence_end():
    assert split_parts('Waited 3.5 hours at the gate') == ['Waited 3.5 hours at the gate']