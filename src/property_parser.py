from hyperrectangles import load_hyperrectangles
import os


def parse_properties(datasets, encoding_models, h_names, target='vnnlib', path='datasets'):
    for dataset in datasets:
        for encoding_model, encoding_name in encoding_models.items():
            for h, h_name in h_names.items():
                if len(h_name) > 1:
                    h_name = ['perturbations']
                h_name = h_name[0]
                
                hyperrectangles = load_hyperrectangles(dataset, encoding_name, h_name, load_saved_hyperrectangles=True)
                print(f'{dataset} -|- {encoding_name} -|-  -|- {hyperrectangles.shape} -|- {h_name}')
                
                properties_directory = f'{path}/{dataset}/properties/{target}/{encoding_name}/{h_name}'
                if not os.path.exists(properties_directory):
                    os.makedirs(properties_directory)

                if target == 'vnnlib':
                    for i, cube in enumerate(hyperrectangles):
                        with open(f'{properties_directory}/prop_{i}_{h_name}.vnnlib', 'w') as property_file:

                            property_file.write(f'; {dataset} {h_name} property.\n\n')
                            for j,d in enumerate(cube):
                                property_file.write(f'(declare-const X_{j} Real)\n')
                            property_file.write(f'\n(declare-const Y_0 Real)\n')
                            property_file.write(f'(declare-const Y_1 Real)\n\n')

                            property_file.write('; Input constraints:\n')
                            for j,d in enumerate(cube):
                                property_file.write(f'(assert (>= X_{j} {d[0]}))\n')
                                property_file.write(f'(assert (<= X_{j} {d[1]}))\n\n')
                            property_file.write('; Output constraints:\n')
                            property_file.write('(assert (<= Y_0 Y_1))')
                
                elif target == 'marabou':
                    for i, cube in enumerate(hyperrectangles):
                        with open(f'{properties_directory}/{h_name}@{i}', 'w') as property_file:
                            for j,d in enumerate(cube):
                                property_file.write(f'x {j} >= {d[0]}\n')
                                property_file.write(f'x {j} <= {d[1]}\n')
                            property_file.write('y0 <= y1')

                else:
                    raise Exception('Target not available. Choose from [vnnlib, marabou].')