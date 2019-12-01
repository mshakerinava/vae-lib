def log(s, disp=True, write=True, file='log.txt', **kwargs):
    if disp:
        print(s, **kwargs)
    if write:
        with open(file=file, mode='a') as f:
            print(s, file=f, **kwargs)


def log_tabular(vals, keys=None, formats=None, file_txt='log.txt', file_csv='log.csv', **kwargs):
    log(','.join([str(x) for x in vals]), disp=False, file=file_csv, **kwargs)
    if formats is not None:
        assert len(formats) == len(vals)
        vals = [x[0] % x[1] for x in zip(formats, vals)]
    if keys is not None:
        assert len(keys) == len(vals)
        log(' | '.join(['%s: %s' % (x[0], str(x[1])) for x in zip(keys, vals)]), file=file_txt, **kwargs)


def clear_logs(file_txt='log.txt', file_csv='log.csv'):
    open(file_txt, 'w').close()
    open(file_csv, 'w').close()
