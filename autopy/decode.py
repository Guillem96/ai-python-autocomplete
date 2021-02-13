def decode_sequence(tokenizer, ids, stop_on_sep=True):
    result = ''
    sep_id = tokenizer.token_to_id('<sep>')
    decoded = tokenizer.decode(ids)

    if sep_id not in decoded:
        decoded = decoded[:decoded.index(sep_id)]

    for w in decoded.split(' '):
        if w.startswith('##'):
            result += w.replace('##', '')
        else:
            result += ' ' + w

    return result
