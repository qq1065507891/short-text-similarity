import GrobalParament
import utrils


def bulid_dict(file_path):
    """
    构建文档字典
    :param file_path:
    :return:
    """

    doc_dict = {}
    with open(file_path, 'r', encoding=GrobalParament.encoding) as f:
        for line in f:
            line = utrils.delete_r_n(line)
            line_list = line.split(",")
            if len(line_list) == 2:
                doc_dict[line_list[0]] = line_list[1]
    return doc_dict


def sim_result_out(sim_out_path, test_dict, train_dict, result_path):
    f_writer = open(result_path, 'w', encoding=GrobalParament.encoding)

    with open(sim_out_path, 'r', encoding=GrobalParament.encoding) as  f_reader:
        for line in f_reader:
            line = utrils.delete_r_n(line)
            line_list = line.split(',')
            if len(line_list) == 2:
                test_docID = line_list[0]
                sim_result = test_docID + ',' + test_dict[test_docID] + '\n' + '-*****最像似的前20个****\n'

                train_docID_list = line_list[1].split()

                for id in train_docID_list:
                    sim_result = sim_result + id + ',' + train_dict[id] + '\n'

                f_writer.write(sim_result)
                f_writer.write("*********************************\n")
                f_writer.flush()
    f_writer.close()


if __name__ == '__main__':
    train_dict = bulid_dict(GrobalParament.train_after_process_text_dir)
    test_dict = bulid_dict(GrobalParament.test_after_process_text_dir)
    sim_result_out(GrobalParament.out_result_path, test_dict, train_dict, GrobalParament.sim_result_path)
