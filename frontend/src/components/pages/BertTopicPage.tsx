import { FC } from 'react';
import { useParams } from 'react-router-dom';
import { useAppContext } from '../../../src/core/context';
import { BertTopicForm } from '../forms/BertTopicForm';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

export const BertTopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject },
  } = useAppContext();

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explore">
      <div className="container">
        <div className="row justify-content-center">
          {JSON.stringify(currentProject ? currentProject.bertopic : null)}
          <BertTopicForm projectSlug={projectName || null} />
        </div>
      </div>
    </ProjectPageLayout>
  );
};
