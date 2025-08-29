import { FC } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useParams } from 'react-router-dom';
import { AnnotationDisagreementManagement } from '../components/AnnotationDisagreementManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { SchemesComparisonManagement } from '../components/SchemesComparisonManagement';
export const CuratePage: FC = () => {
  const { projectName: projectSlug } = useParams();

  if (!projectSlug) {
    return <div>Project not found</div>;
  }

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="curate">
      <div className="container-fluid">
        <div className="row">
          <Tabs id="panel" className="mt-3" defaultActiveKey="scheme">
            <Tab eventKey="scheme" title="Current scheme">
              <AnnotationDisagreementManagement projectSlug={projectSlug} />
            </Tab>
            <Tab eventKey="between" title="Between schemes">
              <SchemesComparisonManagement projectSlug={projectSlug} />
            </Tab>
          </Tabs>
        </div>
      </div>{' '}
    </ProjectPageLayout>
  );
};
