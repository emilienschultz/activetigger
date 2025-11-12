import { FC, useEffect, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useParams } from 'react-router-dom';
import { useAppContext } from '../core/context';

import { useLocation } from 'react-router-dom';
import { AnnotationDisagreementManagement } from '../components/AnnotationDisagreementManagement';
import { AnnotationManagement } from '../components/AnnotationManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { SchemesComparisonManagement } from '../components/SchemesComparisonManagement';

/**
 * Annotation page
 */
export const ProjectTagPage: FC = () => {
  // parameters
  const { projectName } = useParams();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const tab = queryParams.get('tab');
  const {
    appContext: { currentProject: project, displayConfig },
  } = useAppContext();
  const canEdit = displayConfig.interfaceType !== 'annotator';
  const [activeTab, setActiveTab] = useState<string>('tag');
  useEffect(() => {
    setActiveTab(tab || 'tag');
  }, [tab]);

  // nb users
  const nbUsers = project?.users ? Object.keys(project.users).length : 0;

  if (!projectName) return;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="tag">
      {!canEdit || nbUsers < 2 ? (
        <AnnotationManagement />
      ) : (
        <Tabs className="mt-3" activeKey={activeTab} onSelect={(k) => setActiveTab(k || 'tag')}>
          <Tab eventKey="tag" title="Tag">
            <AnnotationManagement />
          </Tab>

          {nbUsers > 1 && (
            <Tab eventKey="curate" title="Curate">
              <Tabs id="panel" className="mt-3" defaultActiveKey="scheme">
                <Tab eventKey="scheme" title="Current scheme">
                  <AnnotationDisagreementManagement projectSlug={projectName} />
                </Tab>
                <Tab eventKey="between" title="Between schemes">
                  <SchemesComparisonManagement projectSlug={projectName} />
                </Tab>
              </Tabs>
            </Tab>
          )}
        </Tabs>
      )}
    </ProjectPageLayout>
  );
};
