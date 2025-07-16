import { FC, useEffect, useState } from 'react';
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

import { reorderLabels } from '../../core/utils';
import { ProjectHistory } from '../ProjectHistory';
import { ProjectParameters } from '../ProjectParameters';
import { SchemesManagement } from '../SchemesManagement';

/**
 * Component to display the project page
 */
export const ProjectPage: FC = () => {
  // get data
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, history, displayConfig },
    setAppContext,
  } = useAppContext();

  // define variables
  const kindScheme =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind
      : '';
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels || []
      : [];

  // sort labels according to the displayConfig
  const availableLabelsSorted = reorderLabels(
    availableLabels as string[],
    displayConfig.labelsOrder || [],
  );

  // redirect if at least 2 tags

  // get the fact that we come from the create page
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [fromProjectPage, setFromProjectPage] = useState<boolean>(false);
  if (!fromProjectPage && searchParams.get('fromProjectPage') === 'true') {
    setFromProjectPage(true);
  }

  // if conditions, navigate to the tag page
  useEffect(() => {
    if (fromProjectPage && availableLabels.length > 1) {
      navigate(`/projects/${projectSlug}/tag`);
      setFromProjectPage(false);
    }
  }, [fromProjectPage, availableLabels.length, navigate, projectSlug]);

  if (!projectSlug || !project) return;

  return (
    <ProjectPageLayout projectName={projectSlug}>
      <div className="container-fluid">
        <Tabs id="panel" className="mt-3" defaultActiveKey="schemes">
          <Tab eventKey="schemes" title="Schemes">
            {availableLabels.length === 0 && (
              <div className="alert alert-info col-12 col-md-8 m-2">
                No labels available for this scheme. Please add labels to use this scheme, or create
                a new scheme.
              </div>
            )}
            <SchemesManagement
              projectSlug={projectSlug}
              deactivateModifications={displayConfig.interfaceType === 'annotator'}
            />
            <LabelsManagement
              projectSlug={projectSlug}
              currentScheme={currentScheme || null}
              availableLabels={availableLabelsSorted as string[]}
              kindScheme={kindScheme as string}
              setAppContext={setAppContext}
              deactivateModifications={displayConfig.interfaceType === 'annotator'}
            />
          </Tab>
          <Tab eventKey="codebook" title="Codebook">
            <CodebookManagement projectName={projectSlug} currentScheme={currentScheme || null} />
          </Tab>

          <Tab eventKey="features" title="Features">
            <FeaturesManagement />
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
