import { FC } from 'react';

import { useStatistics } from '../core/api';

interface StatisticsProps {
  projectSlug: string;
  scheme: string;
}

/**
 * Component to display statistics
 */
export const ProjectStatistics: FC<StatisticsProps> = ({ projectSlug, scheme }) => {
  // get the statistics of the project from the API
  const { statistics } = useStatistics(projectSlug, scheme);
  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-md-12">
          {statistics && (
            <table className="table-statistics">
              <tbody>
                <tr className="table-delimiter">
                  <td>Trainset</td>
                  <td></td>
                </tr>
                <tr>
                  <td>Users involved</td>
                  <td>
                    {statistics['users'].map((e) => (
                      <span key={e} className="m-2">
                        {e}
                      </span>
                    ))}
                  </td>
                </tr>
                <tr>
                  <td>Total</td>
                  <td>{statistics['train_set_n']}</td>
                </tr>
                <tr>
                  <td>Annotated</td>
                  <td>{statistics['train_annotated_n']}</td>
                </tr>

                <tr>
                  <td>Distribution</td>
                  <td>{JSON.stringify(statistics['train_annotated_distribution'])}</td>
                </tr>
                <tr className="table-delimiter">
                  <td>Testset</td>
                  <td></td>
                </tr>
                <tr>
                  <td>Total</td>
                  <td>{statistics['test_set_n']}</td>
                </tr>
                <tr>
                  <td>Annotated</td>
                  <td>{statistics['test_annotated_n']}</td>
                </tr>
                <tr>
                  <td>Distribution</td>
                  <td>{JSON.stringify(statistics['test_annotated_distribution'])}</td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};
