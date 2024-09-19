import { FC } from 'react';
import { VictoryAxis, VictoryBar, VictoryChart, VictoryTheme } from 'victory';

import { useStatistics } from '../core/api';

interface StatisticsProps {
  projectSlug: string;
  scheme: string;
}

interface BarChartProps {
  dataDict: Record<string, number> | null;
}

export const BarChart: FC<BarChartProps> = ({ dataDict }) => {
  if (!dataDict) {
    return null;
  }
  const data = Object.keys(dataDict).map((key) => ({
    x: key,
    y: dataDict[key],
  }));

  return (
    <div style={{ width: '200px', margin: '0 auto' }}>
      <VictoryChart theme={VictoryTheme.material} domainPadding={{ x: 50 }}>
        <VictoryAxis
          style={{
            tickLabels: { fontSize: 12, padding: 5 },
          }}
        />
        <VictoryAxis
          dependentAxis
          label="Number"
          style={{
            axisLabel: { fontSize: 14, padding: 30 },
            tickLabels: { fontSize: 12, padding: 5 },
          }}
        />
        <VictoryBar data={data} style={{ data: { fill: '#4a90e2' } }} barWidth={30} />
      </VictoryChart>
    </div>
  );
};

/**
 * Component to display statistics
 */
export const ProjectStatistics: FC<StatisticsProps> = ({ projectSlug, scheme }) => {
  // get the statistics of the project from the API
  const { statistics } = useStatistics(projectSlug, scheme);

  // graph for the distribution
  //const dataDict = statistics ? statistics['train_annotated_distribution'] : {};

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-md-12">
          {statistics && (
            <div>
              <table className="table-statistics">
                <tbody>
                  <tr className="table-delimiter">
                    <td>Trainset</td>
                    <td></td>
                  </tr>
                  <tr>
                    <td>Users involved</td>
                    <td>
                      <ul>
                        {statistics['users'].map((e) => (
                          <li key={e}>{e}</li>
                        ))}
                      </ul>
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
                    <td>
                      <BarChart dataDict={statistics['train_annotated_distribution']} />
                    </td>
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
                    <td>
                      {' '}
                      <BarChart dataDict={statistics['test_annotated_distribution'] || null} />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
