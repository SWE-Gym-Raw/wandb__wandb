# Generated by ariadne-codegen
# Source: tools/pydantic_codegen/queries-automations.graphql

__all__ = [
    "CREATE_FILTER_TRIGGER_GQL",
    "DELETE_TRIGGER_GQL",
    "SLACK_INTEGRATIONS_FOR_TEAM_GQL",
    "SLACK_INTEGRATIONS_FOR_USER_GQL",
    "TRIGGERS_IN_USER_ORG_ENTITY_GQL",
    "TRIGGERS_IN_USER_TEAMS_GQL",
]

TRIGGERS_IN_USER_ORG_ENTITY_GQL = """
query TriggersInUserOrgEntity($entityName: String, $cursor: String, $perPage: Int) {
  viewer(entityName: $entityName) {
    organizations {
      orgEntity {
        projects(after: $cursor, first: $perPage) {
          ...PaginatedProjectTriggers
        }
      }
    }
  }
}

fragment ArtifactPortfolioScope on ArtifactPortfolio {
  __typename
  id
  name
}

fragment ArtifactSequenceScope on ArtifactSequence {
  __typename
  id
  name
}

fragment FilterEventTriggeringCondition on FilterEventTriggeringCondition {
  eventType
  filter
}

fragment NotificationAction on NotificationTriggeredAction {
  __typename
  title
  message
  severity
  integration {
    ...SlackIntegration
  }
}

fragment PageInfo on PageInfo {
  endCursor
  hasNextPage
}

fragment PaginatedProjectTriggers on ProjectConnection {
  pageInfo {
    ...PageInfo
  }
  edges {
    cursor
    node {
      triggers {
        ...Trigger
      }
    }
  }
}

fragment ProjectScope on Project {
  __typename
  id
  name
}

fragment QueueJobAction on QueueJobTriggeredAction {
  __typename
  template
  queue {
    ...RunQueue
  }
}

fragment RunQueue on RunQueue {
  id
  name
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}

fragment Trigger on Trigger {
  id
  createdAt
  createdBy {
    ...UserInfo
  }
  updatedAt
  name
  description
  enabled
  scope {
    __typename
    ...ProjectScope
    ...ArtifactPortfolioScope
    ...ArtifactSequenceScope
  }
  event: triggeringCondition {
    __typename
    ...FilterEventTriggeringCondition
  }
  action: triggeredAction {
    __typename
    ...QueueJobAction
    ...NotificationAction
    ...WebhookAction
  }
}

fragment UserInfo on User {
  id
  username
}

fragment WebhookAction on GenericWebhookTriggeredAction {
  __typename
  requestPayload
  integration {
    ...WebhookIntegration
  }
}

fragment WebhookIntegration on GenericWebhookIntegration {
  __typename
  id
  name
  urlEndpoint
  secretRef
  accessTokenRef
  createdAt
}
"""

TRIGGERS_IN_USER_TEAMS_GQL = """
query TriggersInUserTeams($entityName: String, $cursor: String, $perPage: Int) {
  viewer(entityName: $entityName) {
    organizations {
      teams {
        projects(after: $cursor, first: $perPage) {
          ...PaginatedProjectTriggers
        }
      }
    }
  }
}

fragment ArtifactPortfolioScope on ArtifactPortfolio {
  __typename
  id
  name
}

fragment ArtifactSequenceScope on ArtifactSequence {
  __typename
  id
  name
}

fragment FilterEventTriggeringCondition on FilterEventTriggeringCondition {
  eventType
  filter
}

fragment NotificationAction on NotificationTriggeredAction {
  __typename
  title
  message
  severity
  integration {
    ...SlackIntegration
  }
}

fragment PageInfo on PageInfo {
  endCursor
  hasNextPage
}

fragment PaginatedProjectTriggers on ProjectConnection {
  pageInfo {
    ...PageInfo
  }
  edges {
    cursor
    node {
      triggers {
        ...Trigger
      }
    }
  }
}

fragment ProjectScope on Project {
  __typename
  id
  name
}

fragment QueueJobAction on QueueJobTriggeredAction {
  __typename
  template
  queue {
    ...RunQueue
  }
}

fragment RunQueue on RunQueue {
  id
  name
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}

fragment Trigger on Trigger {
  id
  createdAt
  createdBy {
    ...UserInfo
  }
  updatedAt
  name
  description
  enabled
  scope {
    __typename
    ...ProjectScope
    ...ArtifactPortfolioScope
    ...ArtifactSequenceScope
  }
  event: triggeringCondition {
    __typename
    ...FilterEventTriggeringCondition
  }
  action: triggeredAction {
    __typename
    ...QueueJobAction
    ...NotificationAction
    ...WebhookAction
  }
}

fragment UserInfo on User {
  id
  username
}

fragment WebhookAction on GenericWebhookTriggeredAction {
  __typename
  requestPayload
  integration {
    ...WebhookIntegration
  }
}

fragment WebhookIntegration on GenericWebhookIntegration {
  __typename
  id
  name
  urlEndpoint
  secretRef
  accessTokenRef
  createdAt
}
"""

CREATE_FILTER_TRIGGER_GQL = """
mutation CreateFilterTrigger($params: CreateFilterTriggerInput!) {
  createFilterTrigger(input: $params) {
    ...CreateFilterTriggerResult
  }
}

fragment ArtifactPortfolioScope on ArtifactPortfolio {
  __typename
  id
  name
}

fragment ArtifactSequenceScope on ArtifactSequence {
  __typename
  id
  name
}

fragment CreateFilterTriggerResult on CreateFilterTriggerPayload {
  __typename
  trigger {
    ...Trigger
  }
  clientMutationId
}

fragment FilterEventTriggeringCondition on FilterEventTriggeringCondition {
  eventType
  filter
}

fragment NotificationAction on NotificationTriggeredAction {
  __typename
  title
  message
  severity
  integration {
    ...SlackIntegration
  }
}

fragment ProjectScope on Project {
  __typename
  id
  name
}

fragment QueueJobAction on QueueJobTriggeredAction {
  __typename
  template
  queue {
    ...RunQueue
  }
}

fragment RunQueue on RunQueue {
  id
  name
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}

fragment Trigger on Trigger {
  id
  createdAt
  createdBy {
    ...UserInfo
  }
  updatedAt
  name
  description
  enabled
  scope {
    __typename
    ...ProjectScope
    ...ArtifactPortfolioScope
    ...ArtifactSequenceScope
  }
  event: triggeringCondition {
    __typename
    ...FilterEventTriggeringCondition
  }
  action: triggeredAction {
    __typename
    ...QueueJobAction
    ...NotificationAction
    ...WebhookAction
  }
}

fragment UserInfo on User {
  id
  username
}

fragment WebhookAction on GenericWebhookTriggeredAction {
  __typename
  requestPayload
  integration {
    ...WebhookIntegration
  }
}

fragment WebhookIntegration on GenericWebhookIntegration {
  __typename
  id
  name
  urlEndpoint
  secretRef
  accessTokenRef
  createdAt
}
"""

DELETE_TRIGGER_GQL = """
mutation DeleteTrigger($id: ID!) {
  deleteTrigger(input: {triggerID: $id}) {
    ...DeleteTriggerResult
  }
}

fragment DeleteTriggerResult on DeleteTriggerPayload {
  __typename
  success
  clientMutationId
}
"""

SLACK_INTEGRATIONS_FOR_USER_GQL = """
query SlackIntegrationsForUser($cursor: String, $perPage: Int) {
  viewer {
    userEntity {
      integrations(after: $cursor, first: $perPage) {
        ...PaginatedIntegrations
      }
    }
  }
}

fragment PageInfo on PageInfo {
  endCursor
  hasNextPage
}

fragment PaginatedIntegrations on IntegrationConnection {
  pageInfo {
    ...PageInfo
  }
  edges {
    cursor
    node {
      __typename
      ...SlackIntegration
    }
  }
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}
"""

SLACK_INTEGRATIONS_FOR_TEAM_GQL = """
query SlackIntegrationsForTeam($entityName: String, $cursor: String, $perPage: Int) {
  entity(name: $entityName) {
    integrations(after: $cursor, first: $perPage) {
      ...PaginatedIntegrations
    }
  }
}

fragment PageInfo on PageInfo {
  endCursor
  hasNextPage
}

fragment PaginatedIntegrations on IntegrationConnection {
  pageInfo {
    ...PageInfo
  }
  edges {
    cursor
    node {
      __typename
      ...SlackIntegration
    }
  }
}

fragment SlackIntegration on SlackIntegration {
  __typename
  id
  teamName
  channelName
}
"""
