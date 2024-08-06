import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { useProject, useStatistics } from '../core/api';
import { useAppContext } from '../core/context';
import { FeaturesManagement } from './forms/FeaturesManagementForm';
import { SchemesManagement } from './forms/SchemesManagementForm';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

interface StatisticsProps {
  projectSlug: string;
  scheme: string;
}

/**
 * Component to display statistics
 */
const DisplayStatistics: FC<StatisticsProps> = ({ projectSlug, scheme }) => {
  const { statistics } = useStatistics(projectSlug, scheme);

  if (!statistics) return null;

  return (
    <div>
      <h3>Statistics</h3>
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
  );
};

/**
 * Component to display the project page
 */

export const ProjectPage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;

  const {
    appContext: { currentScheme },
  } = useAppContext();

  // API get hook provides the project querying the API for us
  // it also handles auth for us making the component code here very clean
  // project can be undefined has at the very first render the API has not yet responded
  // project undefined means the data is not ready yet or there was an error$

  const { project, reFetch } = useProject(projectName); // get project statefrom the API

  return (
    <ProjectPageLayout projectName={projectName}>
      {project && (
        <div>
          <div>
            <SchemesManagement
              available_schemes={Object.keys(project.schemes.available)}
              projectSlug={projectName}
              reFetchProject={reFetch}
            />
          </div>
          <div>
            <FeaturesManagement
              projectSlug={projectName}
              reFetchProject={reFetch}
              availableFeatures={project.features.available}
              possibleFeatures={project.features.options}
            />
          </div>
          {currentScheme && <DisplayStatistics projectSlug={projectName} scheme={currentScheme} />}
        </div>
      )}
    </ProjectPageLayout>
  );
};
