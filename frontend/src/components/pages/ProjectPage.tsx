import { FC } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { useAppContext } from '../../core/context';
import { CodebookManagement } from '../CodeBookManagement';
import { FeaturesManagement } from '../FeaturesManagement';
import { TestSetManagement } from '../forms/TestSetManagement';
import { ImportAnnotations } from '../ImportAnnotations';
import { LabelsManagement } from '../LabelsManagement';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

import { ProjectHistory } from '../ProjectHistory';
import { ProjectParameters } from '../ProjectParameters';
import { SchemesManagement } from '../SchemesManagement';

/**
 * Component to display the project page
 */

export const ProjectPage: FC = () => {
  // get data
  const { projectName } = useParams();
  const projectSlug = projectName;
  const {
    appContext: { currentScheme, currentProject: project, history },
    setAppContext,
  } = useAppContext();

  // define variables
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme]['labels'] || []
      : [];
  const kindScheme =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme]['kind']
      : '';

  // manage redirect if at least 2 tags
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const fromProjectPage = searchParams.get('fromProjectPage') === 'true';
  if (fromProjectPage) {
    if (availableLabels.length > 1) {
      navigate(`/projects/${projectSlug}/tag`);
    }
  }

  if (!projectSlug || !project) return;

  return (
    <ProjectPageLayout projectName={projectSlug}>
      <div className="container-fluid">
        <Tabs id="panel" className="mt-3" defaultActiveKey="schemes">
          <Tab eventKey="schemes" title="Schemes">
            <SchemesManagement projectSlug={projectSlug} />
            <LabelsManagement
              projectSlug={projectSlug}
              currentScheme={currentScheme || null}
              availableLabels={availableLabels as string[]}
              kindScheme={kindScheme as string}
              reFetchCurrentProject={() => {
                setAppContext((prev) => ({ ...prev, currentProject: null }));
              }}
            />
          </Tab>
          <Tab eventKey="features" title="Features">
            <FeaturesManagement />
          </Tab>
          <Tab eventKey="codebook" title="Codebook">
            <CodebookManagement projectName={projectSlug} currentScheme={currentScheme || null} />
          </Tab>
          <Tab eventKey="import" title="Import">
            <div className="explanations">Import data to this project</div>
            <ImportAnnotations
              projectName={project.params.project_slug}
              currentScheme={currentScheme || null}
            />
            <TestSetManagement
              projectSlug={projectSlug}
              currentScheme={currentScheme || ''}
              testSetExist={project?.params.test}
            />
          </Tab>
          <Tab eventKey="parameters" title="Parameters">
            <ProjectParameters project={project} projectSlug={projectSlug} />
          </Tab>
          <Tab eventKey="session" title="Session history">
            <ProjectHistory
              projectSlug={projectSlug}
              history={history}
              setAppContext={setAppContext}
            />
          </Tab>
        </Tabs>
      </div>
    </ProjectPageLayout>
  );
};
