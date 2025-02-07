import { FC } from 'react';
import DataGrid, { Column } from 'react-data-grid';
import { useGetLogs, useGetServer, useStopProcess } from '../../core/api';
import { PageLayout } from '../layout/PageLayout';

interface Computation {
  unique_id: string;
  user: string;
  time: string;
  kind: string;
}

interface Row {
  time: string;
  user: string;
  action: string;
}

export const MonitorPage: FC = () => {
  const { activeProjects, gpu, cpu, memory, disk, reFetchQueueState } = useGetServer(null);
  const { stopProcess } = useStopProcess();
  const { logs } = useGetLogs('all', 500);

  const columns: readonly Column<Row>[] = [
    {
      name: 'Time',
      key: 'time',
      resizable: true,
    },
    {
      name: 'User',
      key: 'user',
      resizable: true,
    },
    {
      name: 'Action',
      key: 'action',
    },
  ];

  return (
    <PageLayout currentPage="monitor">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <h2 className="subtitle">Monitor ressources</h2>

            {JSON.stringify({ gpu, cpu, memory, disk })}

            <h2 className="subtitle">Monitor the active project processes</h2>
            {Object.keys(activeProjects || {}).map((project) => (
              <div key={project}>
                <div>
                  <table>
                    <thead>
                      <tr>
                        <th colSpan={3} className="table-primary text-primary text-center">
                          {project}
                        </th>
                      </tr>
                      <tr>
                        <th>User</th>
                        <th>Time</th>
                        <th>Kind</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeProjects &&
                        Object.values(activeProjects[project] as Computation[]).map((e) => (
                          <tr key={e.unique_id}>
                            <td>{e.user}</td>
                            <td>{e.time}</td>
                            <td>{e.kind}</td>
                            <td>
                              <button
                                onClick={() => {
                                  stopProcess(e.unique_id);
                                  reFetchQueueState();
                                }}
                                className="btn btn-danger"
                              >
                                kill
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}

            <hr />
            <h2 className="subtitle">Recent activity on all projects</h2>
            {logs ? (
              <DataGrid<Row>
                className="fill-grid mt-2"
                columns={columns}
                rows={(logs as unknown as Row[]) || []}
              />
            ) : (
              <div>No rights</div>
            )}
          </div>
        </div>
      </div>{' '}
    </PageLayout>
  );
};
