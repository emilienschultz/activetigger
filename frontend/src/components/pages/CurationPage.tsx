import { FC } from 'react';
import { useParams } from 'react-router-dom';
import { AnnotationDisagreementManagement } from '../AnnotationDisagreementManagement';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

export const CuratePage: FC = () => {
  const { projectName } = useParams();

  if (!projectName) {
    return <div>Project not found</div>;
  }

  return (
    <ProjectPageLayout projectName={projectName} currentAction="curate">
      <div className="container-fluid">
        <div className="row">
          <AnnotationDisagreementManagement projectSlug={projectName} />
        </div>
      </div>{' '}
    </ProjectPageLayout>
  );
};
