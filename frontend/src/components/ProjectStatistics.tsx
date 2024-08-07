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
  const { statistics } = useStatistics(projectSlug, scheme);

  if (!statistics) return null;

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-md-12">
          <div className="subsection">Statistics</div>
          <table className="table-statistics">
            <tbody>
              <tr className="table-delimiter">
                <td>Trainset</td>
                <td></td>
              </tr>
              <tr>
                <td>Total</td>
                <td>{statistics['trainset_n']}</td>
              </tr>
              <tr>
                <td>Annotated</td>
                <td>{statistics['annotated_n']}</td>
              </tr>
              <tr>
                <td>Users involved</td>
                <td>{statistics['users']}</td>
              </tr>
              <tr>
                <td>Distribution</td>
                <td>{JSON.stringify(statistics['annotated_distribution'])}</td>
              </tr>
              <tr className="table-delimiter">
                <td>Testset</td>
                <td></td>
              </tr>
              <tr>
                <td>Total</td>
                <td>{statistics['testset_n']}</td>
              </tr>
            </tbody>
          </table>
          {/*JSON.stringify(statistics, null, 2)*/}
        </div>
      </div>
    </div>
  );
};
