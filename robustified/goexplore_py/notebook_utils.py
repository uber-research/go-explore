
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from goexplore_py.import_ai import *

sns.set_style('whitegrid')


def p(x, y, treatment, t_name, treatment_data):
    x_to_y = defaultdict(list)
    for r in treatment_data[treatment]:
        for k, v in zip(r[x], r[y]):
            k = (k // 50000) * 50000
            x_to_y[k].append(v)

    all_x = []
    all_y = []
    for k, v in x_to_y.items():
        for e in v:
            all_x.append(k)
            all_y.append(e)
    sns.lineplot(all_x, all_y, label=t_name)


def plot(xs, ys, treatments, treatment_data, keys_of_interest, keys_of_interest_dict, pretty_names):
    for x, xname in xs:
        for y, yname in ys:
            plt.figure(figsize=(8, 5))
            for treatment in treatments:
                key = create_key(treatment, keys_of_interest, keys_of_interest_dict)
                name = create_name(treatment, pretty_names)
                p(x, y, key, name, treatment_data)
            plt.title(yname + ' over ' + xname)
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.legend()
            plt.savefig(yname + ' over ' + xname + '.png')
            plt.show()


def gather_treatments(results_folder, keys_of_interest):
    treatment_dict = {}
    for folder in tqdm(sorted(glob.glob(results_folder + '/*'))):
        meta_data = json.load(open(folder + '/kwargs.json'))
        meta_key = []
        for key in keys_of_interest:
            meta_key.append(meta_data[key])
        meta_key_tuple = tuple(meta_key)
        if meta_key_tuple not in treatment_dict:
            treatment_dict[meta_key_tuple] = []
        treatment_dict[meta_key_tuple].append(folder)
    return treatment_dict


def collect_data(treatments, treatment_dict, keys_of_interest, keys_of_interest_dict, pretty_names):
    treatment_data = {}
    for treatment_id in treatments:
        treatment_key = create_key(treatment_id, keys_of_interest, keys_of_interest_dict)
        all_res = []
        print("Loading treatment:", create_name(treatment_id, pretty_names))
        for folder in tqdm(treatment_dict[treatment_key]):
            compute_frames = []
            real_frames = []
            n_found = []
            max_score = []
            n_rooms = []
            n_objects = []
            for f in sorted(glob.glob('%s/*_set.7z' % folder)):
                data = pickle.load(lzma.open(f, 'rb'))
                real, compute = f.split('/')[-1].split('_set.')[0].split('_')
                compute_frames.append(int(compute))
                real_frames.append(int(real))
                n_found.append(len(data))
                max_score.append(max(data[e] for e in data))
                n_rooms.append(len(set((e.level, e.room) for e in data)))
            all_res.append({'compute': compute_frames, 'real': real_frames, 'found': n_found, 'score': max_score,
                            'rooms': n_rooms})
        treatment_data[treatment_key] = all_res
    return treatment_data


def create_key(param_dict, keys_of_interest, keys_of_interest_dict):
    key_proto = []
    for key in keys_of_interest:
        if key in param_dict:
            key_proto.append(param_dict[key])
        else:
            key_proto.append(keys_of_interest_dict[key])
    return tuple(key_proto)


def create_name(param_dict, pretty_names):
    name = ""
    for i, key in enumerate(param_dict.keys()):
        name += pretty_names[key] + " " + str(param_dict[key])
        if i != len(param_dict) - 1:
            name += " "
    return name
