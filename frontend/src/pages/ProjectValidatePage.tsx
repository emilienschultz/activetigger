import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';

/**
 * Component to display the export page
 */
export const ProjectValidatePage: FC = () => {
  const { projectName } = useParams();

  return (
    <ProjectPageLayout projectName={projectName} currentAction="validate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12"></div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
