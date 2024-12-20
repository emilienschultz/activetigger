import { FC } from 'react';
import { useGetQueue, useStopProcess } from '../../core/api';
import { PageLayout } from '../layout/PageLayout';

interface Computation {
  unique_id: string;
  user: string;
  time: string;
  kind: string;
}

export const MonitorPage: FC = () => {
  const { activeProjects, reFetchQueueState } = useGetQueue(null);
  const { stopProcess } = useStopProcess();

  return (
    <PageLayout currentPage="monitor">
      <div className="container-fluid">
        <div className="row">
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
        </div>
      </div>{' '}
    </PageLayout>
  );
};
