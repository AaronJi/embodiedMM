import common_io
import torch
import requests
import os


class CommonIOTableDataset:
    def __init__(self, table_path, selected_cols, num_threads=4, capacity=1024):
        self.table_path = table_path
        self.selected_cols = selected_cols
        self.num_threads = num_threads
        self.capacity = capacity
        self.data_cnt = 0
        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception as e:
            print('get slice id failed', e)
            self.slice_id = 0
            self.slice_count = 1
        self._reader = self._get_reader(
            slice_id=self.slice_id, slice_count=self.slice_count, num_threads=self.num_threads)
        self.row_count = self._reader.get_row_count()
        self.total_row_count = self._get_reader().get_row_count()
        print("table {} slice_id {} row count {} total row count {}".format(
            self.table_path, self.slice_id, self.row_count, self.total_row_count)
        )

    def _get_reader(self, slice_id=0, slice_count=1, num_threads=0):
        import common_io
        while True:
            try:
                reader = common_io.table.TableReader(
                    self.table_path,
                    selected_cols=self.selected_cols,
                    excluded_cols="",
                    slice_id=slice_id,
                    slice_count=slice_count,
                    num_threads=num_threads,
                    capacity=self.capacity)
                break
            except requests.exceptions.ReadTimeout:
                print('Read time out. Retry')
        return reader

    def _seek(self, offset=0):
        try:
            print("slice_id {} seek offset {}".format(self.slice_id, self._reader.start_pos + offset))
            self._reader.seek(offset=self._reader.start_pos + offset)
            self.data_cnt = offset
        except Exception as e:
            print(f'seek error {e}')
            print("slice_id {} seek offset {}".format(self.slice_id, offset))
            self._reader.seek(offset=offset)
            self.data_cnt = offset  # There may be some problems

    def __del__(self):
        self._reader.close()

    def __len__(self):
        return self.row_count

    def get_total_row_count(self):
        return self.total_row_count

    def __getitem__(self, index):
        while True:
            try:
                column_l = self._reader.read(num_records=1, allow_smaller_final_batch=True)[0]
                self.data_cnt += 1
                break
            except common_io.exception.OutOfRangeException:
                print("reach the end of table, start a new reader")
                self.data_cnt = 0
                self._reader = self._get_reader(
                    slice_id=self.slice_id, slice_count=self.slice_count, num_threads=self.num_threads)
                continue
            except Exception as e:
                print(f'read record failed {e}')
                continue
        column_l = [
            item.decode(encoding="utf8", errors="ignore") if type(item) == bytes else item
            for item in column_l
        ]
        return column_l


def parse_table_name(odps_table_name):
    odps_table_name = odps_table_name.split('/')
    table_project = odps_table_name[2]
    table_name = odps_table_name[4]
    if len(odps_table_name) > 5:
        table_partition = odps_table_name[5:]
    else:
        table_partition = None
    return table_project, table_name, table_partition


class AIStudioDataset:
    def __init__(self,
                 table_path, selected_cols, num_threads=4, capacity=1024
                 ):
        self.selected_cols = selected_cols.split(',')
        self.num_threads = num_threads  # https://yuque.antfin-inc.com/pai-user/manual/torch_table 里面有写，但实际上不可用
        self.capacity = capacity  # 同上
        self.table_name = table_path
        self.slice_id = torch.distributed.get_rank()
        self.slice_count = torch.distributed.get_world_size()
        self.get_data_cnt = 0
        self.row_count = 0
        self.total_row_count = 0
        self._reader = self._get_reader()

    def _compute_data_size(self, total_data_size):
        if total_data_size % self.slice_count == 0:
            return total_data_size // self.slice_count
        if self.slice_id < self.slice_count - 1:
            return total_data_size // self.slice_count + 1
        return total_data_size - (total_data_size // self.slice_count + 1) * (self.slice_count - 1)

    def _get_reader(self):
        from pypai.io import TableReader
        from pypai.utils import env_utils

        table_project, table_name, table_partition = parse_table_name(self.table_name)
        project = table_project if table_project else os.environ['ODPS_PROJECT']
        if project == 'graph_embedding' or project == 'graph_embedding_dev':
            endpoint = "http://service-corp.odps.aliyun-inc.com/api"
        else:
            endpoint = "http://service.odps.aliyun-inc.com/api"
        o = env_utils.ODPS(os.environ['ENV_ODPS_ACCESS_ID'], os.environ['ENV_ODPS_ACCESS_KEY'], project=project,
                           endpoint=endpoint)

        reader = TableReader.from_ODPS_type(o,
                                            table_name,
                                            partition=','.join(table_partition),
                                            )
        iterator = reader.to_iterator(batch_size=1,
                                      columns=self.selected_cols, num_worker=self.slice_count,
                                      index_worker=self.slice_id)
        self.total_row_count = reader.table_size
        print('total record num', self.total_row_count)
        self.row_count = self._compute_data_size(self.total_row_count)
        print('slice record num', self.row_count)

        return iterator

    def __getitem__(self, index):
        while True:
            try:
                record = next(self._reader)
                break
            except Exception as e:
                print(f'read failed {e}')
                print("reach the end of table: %d" % self.get_data_cnt)
                self.get_data_cnt = 0
                print("start a new reader")
                self._reader = self._get_reader()

        self.get_data_cnt += 1
        column_l = [
            record[0][i].decode(encoding="utf8", errors="ignore") if type(record[0][i]) == bytes else record[0][i]
            for i in range(len(self.selected_cols))
        ]

        return column_l

    def _seek(self, offset=0):
        pass

    def __del__(self):
        self._reader.close()

    def __len__(self):
        return self.row_count

    def get_total_row_count(self):
        return self.total_row_count


# try:
#     import pypai
#
#     print('use AIstudioReader')
#     TableDataset = AIStudioDataset
# except ImportError:
#     print('use common_io')
#     TableDataset = CommonIOTableDataset

TableDataset = CommonIOTableDataset
