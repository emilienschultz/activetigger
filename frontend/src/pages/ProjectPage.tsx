import { FC, useEffect, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { CodebookManagement } from '../components/CodeBookManagement';
import { FeaturesManagement } from '../components/FeaturesManagement';
import { EvalSetsManagement } from '../components/forms/EvalSetsManagement';
import { ImportAnnotations } from '../components/ImportAnnotations';
import { LabelsManagement } from '../components/LabelsManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useAppContext } from '../core/context';

import { ProjectHistory } from '../components/ProjectHistory';
import { ProjectParameters } from '../components/ProjectParameters';
import { SchemesManagement } from '../components/SchemesManagement';
import { reorderLabels } from '../core/utils';

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
    if (currentScheme && fromProjectPage && availableLabels.length > 1) {
      navigate(`/projects/${projectSlug}/tag`);
      setFromProjectPage(false);
      console.log('fromProjectPage', fromProjectPage, availableLabels, currentScheme);
    }
  }, [fromProjectPage, availableLabels, navigate, projectSlug, currentScheme]);

  if (!projectSlug || !project) return;

  console.log(availableLabels.length);

  return (
    <ProjectPageLayout projectName={projectSlug}>
      <Tabs id="panel" className="mt-3" defaultActiveKey="schemes">
        <Tab eventKey="schemes" title="Schemes">
          <div className="explanations">Manage the schemes and labels</div>
          <SchemesManagement
            projectSlug={projectSlug}
            deactivateModifications={displayConfig.interfaceType === 'annotator'}
          />
          {availableLabels.length === 0 && (
            <div className="alert alert-info col-12 mt-2">
              No labels available for this scheme. Please add labels to use this scheme, or create a
              new scheme.
            </div>
          )}
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
          <div className="explanations">Keep track of the tagging rules</div>
          <CodebookManagement projectName={projectSlug} currentScheme={currentScheme || null} />
        </Tab>

        <Tab eventKey="features" title="Features">
          <div className="explanations">Available features</div>
          <FeaturesManagement />
        </Tab>

        <Tab eventKey="import" title="Import">
          <div className="explanations">Import data to this project</div>
          <ImportAnnotations
            projectName={project.params.project_slug}
            currentScheme={currentScheme || null}
          />
          <EvalSetsManagement
            projectSlug={projectSlug}
            currentScheme={currentScheme || ''}
            dataset={'valid'}
            exist={project?.params.valid}
          />
          <EvalSetsManagement
            projectSlug={projectSlug}
            currentScheme={currentScheme || ''}
            dataset={'test'}
            exist={project?.params.test}
          />
        </Tab>

        <Tab eventKey="parameters" title="Parameters">
          <ProjectParameters project={project} projectSlug={projectSlug} />
        </Tab>

        <Tab eventKey="session" title="Session history">
          <div className="explanations">History of the current session</div>
          <ProjectHistory
            projectSlug={projectSlug}
            history={history}
            setAppContext={setAppContext}
          />
        </Tab>
      </Tabs>
    </ProjectPageLayout>
  );
};
