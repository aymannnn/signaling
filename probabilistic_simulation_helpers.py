
def _quartile_to_id_list(n) -> dict:
    quartile_size = n/4
    quartiles = {
        1: [i for i in range(0, int(quartile_size))],
        2: [i for i in range(int(quartile_size), int(quartile_size*2))],
        3: [i for i in range(int(quartile_size*2), int(quartile_size*3))],
        4: [i for i in range(int(quartile_size*3), n)]
    }
    return quartiles

def _id_to_quartile_dict(n) -> dict:
    quartile_size = n/4
    id_to_quartile = {}
    for i in range(n):
        if i < quartile_size:
            id_to_quartile[i] = 1
        elif i < 2*quartile_size:
            id_to_quartile[i] = 2
        elif i < 3*quartile_size:
            id_to_quartile[i] = 3
        else:
            id_to_quartile[i] = 4
    return id_to_quartile

def _decile_to_id_list(n) -> dict:
    decile_size = n/10
    deciles = {
        1: [i for i in range(0, int(decile_size))],
        2: [i for i in range(int(decile_size), int(decile_size*2))],
        3: [i for i in range(int(decile_size*2), int(decile_size*3))],
        4: [i for i in range(int(decile_size*3), int(decile_size*4))],
        5: [i for i in range(int(decile_size*4), int(decile_size*5))],
        6: [i for i in range(int(decile_size*5), int(decile_size*6))],
        7: [i for i in range(int(decile_size*6), int(decile_size*7))],
        8: [i for i in range(int(decile_size*7), int(decile_size*8))],
        9: [i for i in range(int(decile_size*8), int(decile_size*9))],
        10: [i for i in range(int(decile_size*9), n)]
    }
    return deciles

def _id_to_decile_dict(n) -> dict:
    decile_size = n/10
    id_to_decile = {}
    for i in range(n):
        if i < decile_size:
            id_to_decile[i] = 1
        elif i < 2*decile_size:
            id_to_decile[i] = 2
        elif i < 3*decile_size:
            id_to_decile[i] = 3
        elif i < 4*decile_size:
            id_to_decile[i] = 4
        elif i < 5*decile_size:
            id_to_decile[i] = 5
        elif i < 6*decile_size:
            id_to_decile[i] = 6
        elif i < 7*decile_size:
            id_to_decile[i] = 7
        elif i < 8*decile_size:
            id_to_decile[i] = 8
        elif i < 9*decile_size:
            id_to_decile[i] = 9
        else:
            id_to_decile[i] = 10
    return id_to_decile


def get_lookups(sim_settings):
    # quartiles
    program_quartile_to_id_list = _quartile_to_id_list(
        sim_settings['n_programs'])
    program_id_to_quartile = _id_to_quartile_dict(
        sim_settings['n_programs'])
    applicant_quartile_to_id_list = _quartile_to_id_list(
        sim_settings['n_applicants'])
    applicant_id_to_quartile = _id_to_quartile_dict(
        sim_settings['n_applicants'])

    # deciles
    program_decile_to_id_list = _decile_to_id_list(
        sim_settings['n_programs'])
    program_id_to_decile = _id_to_decile_dict(
        sim_settings['n_programs'])
    applicant_decile_to_id_list = _decile_to_id_list(
        sim_settings['n_applicants'])
    applicant_id_to_decile = _id_to_decile_dict(
        sim_settings['n_applicants'])

    lookups = {
        'program_quartile_to_id_list': program_quartile_to_id_list,
        'program_id_to_quartile': program_id_to_quartile,
        'applicant_quartile_to_id_list': applicant_quartile_to_id_list,
        'applicant_id_to_quartile': applicant_id_to_quartile,
        'program_decile_to_id_list': program_decile_to_id_list,
        'program_id_to_decile': program_id_to_decile,
        'applicant_decile_to_id_list': applicant_decile_to_id_list,
        'applicant_id_to_decile': applicant_id_to_decile
    }

    return lookups


def get_dtypes(columns: list) -> dict:
    '''
    Given a list of columns, identify the first as a string and the
    remainder as booleans for later use in a dataframe read.
    '''
    return {
        **{columns[0]: 'str'},
        **{col: 'bool' for col in columns[1:]}
    }
